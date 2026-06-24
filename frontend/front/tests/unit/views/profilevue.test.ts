import { describe, it, expect } from 'vitest'
import { reactive, ref } from 'vue'

function makeProfileForm(initial = { username: 'testuser', email: 'test@example.com' }) {
  const form = reactive({ username: initial.username, email: initial.email })
  const errors = reactive({ username: '', email: '' })
  const success = ref(false)
  const error = ref('')

  function validate(): boolean {
    errors.username = ''
    errors.email = ''
    success.value = false
    error.value = ''

    if (!form.username || form.username.length < 3) {
      errors.username = 'Username must be at least 3 characters'
    }

    return !errors.username && !errors.email
  }

  return { form, errors, success, error, validate }
}

function makePasswordForm() {
  const form = reactive({ old_password: '', new_password: '', confirm: '' })
  const errors = reactive({ old_password: '', new_password: '', confirm: '' })
  const success = ref(false)
  const error = ref('')

  function validate(): boolean {
    errors.old_password = ''
    errors.new_password = ''
    errors.confirm = ''
    success.value = false
    error.value = ''

    if (!form.old_password) {
      errors.old_password = 'Required'
    }

    if (!form.new_password || form.new_password.length < 8) {
      errors.new_password = 'Min 8 characters'
    } else if (!/[A-Z]/.test(form.new_password)) {
      errors.new_password = 'Must contain uppercase letter'
    } else if (!/[0-9]/.test(form.new_password)) {
      errors.new_password = 'Must contain a number'
    }

    if (!form.confirm || form.new_password !== form.confirm) {
      errors.confirm = 'Passwords do not match'
    }

    return !errors.old_password && !errors.new_password && !errors.confirm
  }

  return { form, errors, success, error, validate }
}

describe('ProfileView: profile update validate()', () => {
  describe('username', () => {
    it('fails when username is empty', () => {
      const { form, errors, validate } = makeProfileForm()
      form.username = ''

      expect(validate()).toBe(false)
      expect(errors.username).toBe('Username must be at least 3 characters')
    })

    it('fails when username is shorter than 3 characters', () => {
      const { form, errors, validate } = makeProfileForm()
      form.username = 'ab'

      expect(validate()).toBe(false)
      expect(errors.username).toBe('Username must be at least 3 characters')
    })

    it('passes with exactly 3 characters', () => {
      const { form, errors, validate } = makeProfileForm()
      form.username = 'abc'

      expect(validate()).toBe(true)
      expect(errors.username).toBe('')
    })

    it('passes with valid username', () => {
      const { errors, validate } = makeProfileForm()

      expect(validate()).toBe(true)
      expect(errors.username).toBe('')
    })

    it('clears previous username error on re-validation', () => {
      const { form, errors, validate } = makeProfileForm()
      form.username = ''
      validate()

      form.username = 'validuser'
      validate()

      expect(errors.username).toBe('')
    })
  })
})

describe('ProfileView: change password validate()', () => {
  describe('old_password', () => {
    it('fails when current password is empty', () => {
      const { errors, validate } = makePasswordForm()

      expect(validate()).toBe(false)
      expect(errors.old_password).toBe('Required')
    })

    it('passes old_password check when it is filled', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'NewPass1'
      form.confirm = 'NewPass1'

      expect(validate()).toBe(true)
      expect(errors.old_password).toBe('')
    })
  })

  describe('new_password', () => {
    it('fails when new password is empty', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'

      expect(validate()).toBe(false)
      expect(errors.new_password).toBe('Min 8 characters')
    })

    it('fails when new password is shorter than 8 characters', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'Ab1'
      form.confirm = 'Ab1'

      expect(validate()).toBe(false)
      expect(errors.new_password).toBe('Min 8 characters')
    })

    it('fails when new password has no uppercase letter', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'alllower1'
      form.confirm = 'alllower1'

      expect(validate()).toBe(false)
      expect(errors.new_password).toBe('Must contain uppercase letter')
    })

    it('fails when new password has no digit', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'NoDigitsHere'
      form.confirm = 'NoDigitsHere'

      expect(validate()).toBe(false)
      expect(errors.new_password).toBe('Must contain a number')
    })

    it('passes with valid new password', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'NewPass1'
      form.confirm = 'NewPass1'

      expect(validate()).toBe(true)
      expect(errors.new_password).toBe('')
    })
  })

  describe('confirm', () => {
    it('fails when confirm is empty', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'NewPass1'

      expect(validate()).toBe(false)
      expect(errors.confirm).toBe('Passwords do not match')
    })

    it('fails when confirm does not match new password', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'NewPass1'
      form.confirm = 'Different1'

      expect(validate()).toBe(false)
      expect(errors.confirm).toBe('Passwords do not match')
    })

    it('passes when confirm matches new password', () => {
      const { form, errors, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'NewPass1'
      form.confirm = 'NewPass1'

      expect(validate()).toBe(true)
      expect(errors.confirm).toBe('')
    })
  })

  describe('combined', () => {
    it('returns false and sets all errors when form is completely empty', () => {
      const { errors, validate } = makePasswordForm()

      expect(validate()).toBe(false)
      expect(errors.old_password).toBeTruthy()
      expect(errors.new_password).toBeTruthy()
      expect(errors.confirm).toBeTruthy()
    })

    it('returns true when all fields are valid', () => {
      const { form, validate } = makePasswordForm()
      form.old_password = 'OldPass1'
      form.new_password = 'NewPass1'
      form.confirm = 'NewPass1'

      expect(validate()).toBe(true)
    })

    it('clears previous errors on re-validation', () => {
      const { form, errors, validate } = makePasswordForm()

      validate()

      form.old_password = 'OldPass1'
      form.new_password = 'NewPass1'
      form.confirm = 'NewPass1'
      validate()

      expect(errors.old_password).toBe('')
      expect(errors.new_password).toBe('')
      expect(errors.confirm).toBe('')
    })
  })
})